function Code(el)
  if FORMAT:match('latex') then
    return pandoc.RawInline('latex', '\\nolinkurl{' .. el.text .. '}')
  end
end

function Para(el)
  if FORMAT:match('latex') and pandoc.utils.stringify(el):match('^Author: Timothy') then
    return {pandoc.RawBlock('latex', '\\clearpage'), el}
  end
end
